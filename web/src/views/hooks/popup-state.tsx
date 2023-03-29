import React from 'react';
import { create } from 'zustand';

interface PopupManagerState {
  overlayActive: boolean,
  activePopupId: string,

  data: object,

  hidePopup() : void;
  activatePopup(id: string, newData?: object) : void;
}

const usePopupStore = create<PopupManagerState>(set => ({
  overlayActive: false,
  activePopupId: null,
  data: {},

  hidePopup: () => set(state => ({
    overlayActive: false,
    activePopupId: null,
  })),

  activatePopup: (id: string, newData?: object) => set(state => ({
    overlayActive: true,
    activePopupId: id,
    data: newData || {},
  })),
}));

const withPopupStore = BaseComponent => props => {
  const store = usePopupStore();
  return (
    <BaseComponent {...props} store={store} />
  );
};

export { PopupManagerState, usePopupStore, withPopupStore };
