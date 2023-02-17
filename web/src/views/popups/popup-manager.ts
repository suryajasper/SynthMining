import m from 'mithril';

import { Popup, PopupAttrs, updatePopupOverlayStatus } from '../popups/popup';

type PopupRecord = Record<string, {
  view: any,
  attrs: PopupAttrs,
}>

export default class PopupManager {
  public popups: PopupRecord;
  private reloadCallback?: () => void;

  constructor() {
    this.popups = {};
  }

  setReloadCallback(reloadCallback: () => void) {
    this.reloadCallback = reloadCallback;
  }

  public linkPopup(
    popupName: string, popupView: any
  ): void {
    if (!this.reloadCallback) 
      throw new Error("Reload Callback Not Initialized");

    const attrs : PopupAttrs = {
      active: true,
      data: undefined,

      disabledCallback: this.inactivateAllPopups.bind(this),
      reloadCallback: this.reloadCallback,
    };

    this.popups[popupName] = {
      view: popupView,
      attrs,
    }
  }

  isPopupActive() : boolean {
    return Object.values(this.popups)
      .map(popup => popup.attrs.active)
      .reduce((a, b) => a || b);
  }

  inactivateAllPopups() : void {
    updatePopupOverlayStatus({ active: false });

    Object.values(this.popups).forEach(popup => {
      popup.attrs.active = false;
    })
  }

  updatePopupStatus({ name, active, data }) : void {
    if (!this.popups[name]) 
      return;
    
    this.inactivateAllPopups();
    updatePopupOverlayStatus({ active });

    this.popups[name].attrs.active = active;
    this.popups[name].attrs.data = data;

    m.redraw();
  }
}