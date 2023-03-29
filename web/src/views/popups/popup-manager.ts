import { PopupAttrs } from '../popups/popup';

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
      data: undefined,
      
      reloadCallback: this.reloadCallback,
    };

    this.popups[popupName] = {
      view: popupView,
      attrs,
    }
  }
}